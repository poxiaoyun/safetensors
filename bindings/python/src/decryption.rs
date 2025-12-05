//! 信封加密解密模块 (AES-256-CTR + KMS)
//!
//! 加密文件格式:
//! - 4 字节: 元数据长度 (big-endian)
//! - 元数据: JSON {"encryptedKey": "...", "password": "..."}
//! - 16 字节: IV (用于 AES-256-CTR)
//! - 剩余部分: 加密后的内容 (支持随机访问解密)

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

/// 元数据长度字段大小 (4 字节)
const METADATA_LENGTH_SIZE: usize = 4;
/// IV 大小 (16 字节，用于 AES-256-CTR)
const IV_SIZE: usize = 16;

/// 信封加密文件的元数据
#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
struct EnvelopeMetadata {
    encrypted_key: String,
    password: String,
}

/// 加密源，包含文件句柄和解密所需的密钥/IV
pub struct EncryptedSource {
    pub file: Mutex<File>,
    pub key: Vec<u8>,
    pub iv: Vec<u8>,
    /// 数据区域在文件中的起始偏移 (元数据长度 + 元数据 + IV 之后)
    pub data_offset: usize,
}

/// KMS 客户端配置
struct KmsClient {
    base_url: String,
}

impl KmsClient {
    fn new() -> Self {
        let base_url = env::var("KMS_BASE_URL")
            .unwrap_or_else(|_| "https://rune-api.develop.xiaoshiai.cn".to_string());
        Self { base_url }
    }

    /// 调用 KMS 解密数据密钥
    fn decrypt_data_key(&self, password: &str, encrypted_key: &str) -> Result<Vec<u8>, String> {
        let url = format!("{}/api/kms/decrypt-data-key", self.base_url);

        let client = reqwest::blocking::Client::new();
        let resp = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "password": password,
                "encryptedKey": encrypted_key
            }))
            .send()
            .map_err(|e| format!("KMS request failed: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!("KMS returned error: {}", resp.status()));
        }

        let data: serde_json::Value = resp.json()
            .map_err(|e| format!("Failed to parse KMS response: {e}"))?;

        let plaintext_key_b64 = data["plaintextKey"]
            .as_str()
            .ok_or("Missing plaintextKey in KMS response")?;

        base64_decode(plaintext_key_b64)
            .map_err(|e| format!("Failed to decode plaintext key: {e}"))
    }
}

fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    STANDARD.decode(s).map_err(|e| e.to_string())
}

/// 读取信封加密文件的元数据（不解密内容）
fn read_envelope_metadata(file: &mut File) -> Result<Option<(EnvelopeMetadata, Vec<u8>, usize)>, String> {
    // 读取元数据长度 (4 字节, big-endian)
    let mut length_bytes = [0u8; METADATA_LENGTH_SIZE];
    if file.read_exact(&mut length_bytes).is_err() {
        return Ok(None);
    }

    let metadata_len = u32::from_be_bytes(length_bytes) as usize;

    // 合理性检查
    if metadata_len < 10 || metadata_len > 10 * 1024 {
        return Ok(None);
    }

    // 读取元数据
    let mut metadata_bytes = vec![0u8; metadata_len];
    if file.read_exact(&mut metadata_bytes).is_err() {
        return Ok(None);
    }

    // 解析元数据 JSON
    let metadata: EnvelopeMetadata = match serde_json::from_slice(&metadata_bytes) {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };

    // 读取 IV (16 字节)
    let mut iv = vec![0u8; IV_SIZE];
    if file.read_exact(&mut iv).is_err() {
        return Ok(None);
    }

    // 计算数据起始偏移
    let data_offset = METADATA_LENGTH_SIZE + metadata_len + IV_SIZE;

    Ok(Some((metadata, iv, data_offset)))
}

/// 尝试创建加密源
///
/// 如果文件是信封加密格式，返回 Some((header_size, metadata, source))
/// 否则返回 None
pub fn create_encrypted_source(filename: &Path) -> PyResult<Option<(usize, Metadata, EncryptedSource)>> {
    let mut file = File::open(filename).map_err(|_| {
        PyFileNotFoundError::new_err(format!(
            "No such file or directory: {}",
            filename.display()
        ))
    })?;

    // 尝试读取信封加密元数据
    let (envelope_meta, iv, data_offset) = match read_envelope_metadata(&mut file) {
        Ok(Some(result)) => result,
        Ok(None) => return Ok(None),  // 不是信封加密格式
        Err(e) => {
            return Err(SafetensorError::new_err(format!("Error reading envelope metadata: {e}")));
        }
    };

    // 调用 KMS 解密数据密钥
    let kms = KmsClient::new();
    let key = kms.decrypt_data_key(&envelope_meta.password, &envelope_meta.encrypted_key)
        .map_err(|e| SafetensorError::new_err(format!("KMS decryption failed: {e}")))?;

    if key.len() != 32 {
        return Err(SafetensorError::new_err(format!(
            "Invalid key length: expected 32 bytes, got {}",
            key.len()
        )));
    }

    // 读取 safetensors header size (8 字节, little-endian, 加密的)
    let mut size_buffer = [0u8; 8];
    file.read_exact(&mut size_buffer).map_err(|e| {
        SafetensorError::new_err(format!("Error reading size: {e}"))
    })?;

    // 解密 size
    let mut cipher = Aes256Ctr::new_from_slices(&key, &iv).map_err(|e| {
        SafetensorError::new_err(format!("Error creating cipher: {e}"))
    })?;
    cipher.apply_keystream(&mut size_buffer);

    let n = u64::from_le_bytes(size_buffer) as usize;

    // 验证 header 大小
    let file_size = file.metadata().map_err(|e| {
        SafetensorError::new_err(format!("Error getting file metadata: {e}"))
    })?.len();

    if n as u64 > 100 * 1024 * 1024 || n as u64 > file_size {
        return Err(SafetensorError::new_err(format!(
            "Header too large ({} bytes). Decryption may have failed due to incorrect key.",
            n
        )));
    }

    // 读取并解密 header
    let mut header_buffer = vec![0u8; n];
    file.read_exact(&mut header_buffer).map_err(|e| {
        SafetensorError::new_err(format!("Error reading header: {e}"))
    })?;

    // cipher 状态自动继续
    cipher.apply_keystream(&mut header_buffer);

    // 解析 safetensors metadata
    let metadata: Metadata = serde_json::from_slice(&header_buffer).map_err(|e| {
        SafetensorError::new_err(format!("Error deserializing header: {e}"))
    })?;

    let source = EncryptedSource {
        file: Mutex::new(file),
        key,
        iv,
        data_offset,
    };

    Ok(Some((n, metadata, source)))
}

/// 解密连续的字节范围
///
/// start: 相对于加密数据区域的偏移（不包含信封元数据）
/// len: 要读取的字节数
pub fn decrypt_buffer(source: &EncryptedSource, start: usize, len: usize) -> PyResult<Vec<u8>> {
    let mut buffer = vec![0u8; len];

    // 文件中的实际位置 = data_offset + start
    let file_offset = (source.data_offset + start) as u64;
    // cipher 的 seek 位置是相对于加密开始的位置（即 data_offset 之后）
    let cipher_offset = start as u64;

    {
        let mut file = source.file.lock().map_err(|_| {
            SafetensorError::new_err("Failed to lock file")
        })?;
        file.seek(SeekFrom::Start(file_offset)).map_err(|e| {
            SafetensorError::new_err(format!("Error seeking file: {e}"))
        })?;
        file.read_exact(&mut buffer).map_err(|e| {
            SafetensorError::new_err(format!("Error reading file: {e}"))
        })?;
    }

    let mut cipher = Aes256Ctr::new_from_slices(&source.key, &source.iv).map_err(|e| {
        SafetensorError::new_err(format!("Error creating cipher: {e}"))
    })?;
    cipher.try_seek(cipher_offset).map_err(|e| {
        SafetensorError::new_err(format!("Error seeking cipher: {e}"))
    })?;
    cipher.apply_keystream(&mut buffer);

    Ok(buffer)
}

/// 解密多个不连续的字节范围（用于切片操作）
///
/// tensor_start_offset: 张量数据相对于加密数据区域的偏移
/// indices: 要读取的字节范围列表 [(start, stop), ...]
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
            // 文件中的实际位置
            let file_offset = (source.data_offset + tensor_start_offset + start) as u64;
            // cipher 的 seek 位置（相对于加密开始）
            let cipher_offset = (tensor_start_offset + start) as u64;

            file.seek(SeekFrom::Start(file_offset)).map_err(|e| {
                SafetensorError::new_err(format!("Error seeking file: {e}"))
            })?;

            file.read_exact(&mut buffer[buffer_offset..buffer_offset + len]).map_err(|e| {
                SafetensorError::new_err(format!("Error reading file: {e}"))
            })?;

            // 解密这个块
            let mut cipher = Aes256Ctr::new_from_slices(&source.key, &source.iv).map_err(|e| {
                SafetensorError::new_err(format!("Error creating cipher: {e}"))
            })?;

            cipher.try_seek(cipher_offset).map_err(|e| {
                SafetensorError::new_err(format!("Error seeking cipher: {e}"))
            })?;

            cipher.apply_keystream(&mut buffer[buffer_offset..buffer_offset + len]);

            buffer_offset += len;
        }
    }

    Ok(buffer)
}
