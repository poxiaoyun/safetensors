use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;
use std::env;
use aes::cipher::{KeyIvInit, StreamCipher, StreamCipherSeek};
use aes::Aes256;
use ctr::Ctr128BE;
use serde::Deserialize;
use pyo3::exceptions::PyFileNotFoundError;
use pyo3::prelude::*;
use safetensors::tensor::Metadata;
use crate::SafetensorError;

type Aes256Ctr = Ctr128BE<Aes256>;

#[derive(Deserialize, Clone)]
struct EncryptedFile {
    file: String,
    key: String,
    iv: String,
}

pub struct EncryptedSource {
    pub file: Mutex<File>,
    pub key: Vec<u8>,
    pub iv: Vec<u8>,
}

#[derive(Deserialize, Clone)]
struct EncryptionConfig {
    files: Vec<EncryptedFile>,
}

static GLOBAL_CONFIG: Mutex<Option<EncryptionConfig>> = Mutex::new(None);

pub fn load_global_config_from_json(json: &str) -> Result<(), serde_json::Error> {
    let config: EncryptionConfig = serde_json::from_str(json)?;
    if let Ok(mut guard) = GLOBAL_CONFIG.lock() {
        *guard = Some(config);
    }
    Ok(())
}

pub fn create_encrypted_source(filename: &Path) -> PyResult<Option<(usize, Metadata, EncryptedSource)>> {
    let mut config_opt = {
        if let Ok(guard) = GLOBAL_CONFIG.lock() {
            guard.clone()
        } else {
            None
        }
    };

    if config_opt.is_none() {
        let encryption_config = env::var("SAFETENSORS_ENC_CONFIG").ok();
        if let Some(config_path) = encryption_config {
            if let Ok(config_content) = std::fs::read_to_string(config_path) {
                if let Ok(config) = serde_json::from_str::<EncryptionConfig>(&config_content) {
                    config_opt = Some(config);
                }
            }
        }
    }
    
    if let Some(config) = config_opt {
        let filename_str = filename.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if let Some(file_config) = config.files.iter().find(|f| f.file == filename_str || filename.ends_with(&f.file)) {
                    let mut file = File::open(&filename).map_err(|_| {
                        PyFileNotFoundError::new_err(format!(
                            "No such file or directory: {}",
                            filename.display()
                        ))
                    })?;
                    
                    let key = hex::decode(&file_config.key).map_err(|e| {
                        SafetensorError::new_err(format!("Error decoding key: {e}"))
                    })?;
                    let iv = hex::decode(&file_config.iv).map_err(|e| {
                        SafetensorError::new_err(format!("Error decoding iv: {e}"))
                    })?;

                    // Read size (8 bytes)
                    let mut size_buffer = [0u8; 8];
                    file.read_exact(&mut size_buffer).map_err(|e| {
                        SafetensorError::new_err(format!("Error reading size: {e}"))
                    })?;

                    let mut cipher = Aes256Ctr::new_from_slices(&key, &iv).map_err(|e| {
                        SafetensorError::new_err(format!("Error creating cipher: {e}"))
                    })?;
                    cipher.apply_keystream(&mut size_buffer);

                    let n = u64::from_le_bytes(size_buffer) as usize;

                    // Validate header size
                    // 1. Check against a reasonable absolute limit (e.g., 100MB)
                    // 2. Check against file size
                    let file_size = file.metadata().map_err(|e| {
                        SafetensorError::new_err(format!("Error getting file metadata: {e}"))
                    })?.len();

                    if n as u64 > 100 * 1024 * 1024 || n as u64 > file_size {
                         return Err(SafetensorError::new_err(format!(
                            "Header too large ({} bytes). This likely means decryption failed due to incorrect key/IV, or the file is not actually encrypted.",
                            n
                        )));
                    }

                    // Read header
                    let mut header_buffer = vec![0u8; n];
                    file.read_exact(&mut header_buffer).map_err(|e| {
                        SafetensorError::new_err(format!("Error reading header: {e}"))
                    })?;
                    
                    // Decrypt header (cipher state continues from size)
                    cipher.apply_keystream(&mut header_buffer);

                    let metadata: Metadata = serde_json::from_slice(&header_buffer).map_err(|e| {
                            SafetensorError::new_err(format!("Error deserializing header: {e}"))
                    })?;
                    
                    let source = EncryptedSource {
                        file: Mutex::new(file),
                        key,
                        iv,
                    };
                    
                    return Ok(Some((n, metadata, source)));
        }
    }
    Ok(None)
}

pub fn decrypt_buffer(source: &EncryptedSource, start: usize, len: usize) -> PyResult<Vec<u8>> {
    let mut buffer = vec![0u8; len];
    
    {
        let mut file = source.file.lock().map_err(|_| {
            SafetensorError::new_err("Failed to lock file")
        })?;
        file.seek(SeekFrom::Start(start as u64)).map_err(|e| {
            SafetensorError::new_err(format!("Error seeking file: {e}"))
        })?;
        file.read_exact(&mut buffer).map_err(|e| {
            SafetensorError::new_err(format!("Error reading file: {e}"))
        })?;
    }
    
    let mut cipher = Aes256Ctr::new_from_slices(&source.key, &source.iv).map_err(|e| {
        SafetensorError::new_err(format!("Error creating cipher: {e}"))
    })?;
    cipher.try_seek(start as u64).map_err(|e| {
        SafetensorError::new_err(format!("Error seeking cipher: {e}"))
    })?;
    cipher.apply_keystream(&mut buffer);

    Ok(buffer)
}

pub fn decrypt_slices(source: &EncryptedSource, tensor_start_offset: usize, indices: Vec<(usize, usize)>) -> PyResult<Vec<u8>> {
    let total_len: usize = indices.iter().map(|(start, stop)| stop - start).sum();
    
    let mut buffer = vec![0u8; total_len];
    let mut buffer_offset = 0;
    
    {
        let mut file = source.file.lock().map_err(|_| {
            SafetensorError::new_err("Failed to lock file")
        })?;
        
        for (start, stop) in indices {
            let len = stop - start;
            let file_offset = (tensor_start_offset + start) as u64;
            
            file.seek(SeekFrom::Start(file_offset)).map_err(|e| {
                SafetensorError::new_err(format!("Error seeking file: {e}"))
            })?;
            
            file.read_exact(&mut buffer[buffer_offset..buffer_offset + len]).map_err(|e| {
                SafetensorError::new_err(format!("Error reading file: {e}"))
            })?;
            
            // Decrypt this chunk
            let mut cipher = Aes256Ctr::new_from_slices(&source.key, &source.iv).map_err(|e| {
                SafetensorError::new_err(format!("Error creating cipher: {e}"))
            })?;
            
            cipher.try_seek(file_offset).map_err(|e| {
                SafetensorError::new_err(format!("Error seeking cipher: {e}"))
            })?;
            
            cipher.apply_keystream(&mut buffer[buffer_offset..buffer_offset + len]);
            
            buffer_offset += len;
        }
    }
    
    Ok(buffer)
}
