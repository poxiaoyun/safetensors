//! 信封加密解密模块 (AES / SM4 + KMS)
//!
//! 加密文件格式:
//! - 4 字节: 元数据长度 (big-endian)
//! - 元数据: JSON {"encryptedKey": "...", "algorithm": "AES" | "SM4"}
//! - 16 字节: IV (用于 CTR 模式)
//! - 剩余部分: 加密后的内容 (支持随机访问解密)
//!
//! 密码从环境变量 XPAI_ENC_PASSWORD 获取

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;
use std::env;
use aes::cipher::{KeyIvInit, StreamCipher, StreamCipherSeek};
use aes::Aes256;
use sm4::Sm4;
use ctr::Ctr128BE;
use serde::Deserialize;
use pyo3::exceptions::PyFileNotFoundError;
use pyo3::prelude::*;
use safetensors::tensor::Metadata;
use crate::SafetensorError;

type AesCtr = Ctr128BE<Aes256>;
type Sm4Ctr = Ctr128BE<Sm4>;

/// 支持的加密算法
#[derive(Clone, Debug, PartialEq)]
enum Algorithm {
    AES,
    SM4,
}

impl Algorithm {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "AES" => Some(Algorithm::AES),
            "SM4" => Some(Algorithm::SM4),
            _ => None,
        }
    }
}

/// 元数据长度字段大小 (4 字节)
const METADATA_LENGTH_SIZE: usize = 4;
/// IV 大小 (16 字节，用于 CTR 模式)
const IV_SIZE: usize = 16;

/// 信封加密文件的元数据
#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
struct EnvelopeMetadata {
    encrypted_key: String,
    algorithm: String,
}

/// 加密源，包含文件句柄和解密所需的密钥/IV/算法
pub struct EncryptedSource {
    pub file: Mutex<File>,
    pub key: Vec<u8>,
    pub iv: Vec<u8>,
    /// 数据区域在文件中的起始偏移 (元数据长度 + 元数据 + IV 之后)
    pub data_offset: usize,
    /// 加密算法
    pub algorithm: Algorithm,
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
    fn decrypt_data_key(&self, encrypted_key: &str, algorithm: &str) -> Result<Vec<u8>, String> {
        // 从环境变量获取密码
        let password = env::var("XPAI_ENC_PASSWORD")
            .map_err(|_| "XPAI_ENC_PASSWORD environment variable not set")?;

        let url = format!("{}/api/kms/decrypt-data-key", self.base_url);

        let client = reqwest::blocking::Client::new();
        let resp = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "password": password,
                "encryptedKey": encrypted_key,
                "algorithm": algorithm
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

    // 解析算法
    let algorithm = Algorithm::from_str(&envelope_meta.algorithm)
        .ok_or_else(|| SafetensorError::new_err(format!(
            "Unsupported algorithm: {}. Supported: AES, SM4",
            envelope_meta.algorithm
        )))?;

    // 调用 KMS 解密数据密钥
    let kms = KmsClient::new();
    let key = kms.decrypt_data_key(&envelope_meta.encrypted_key, &envelope_meta.algorithm)
        .map_err(|e| SafetensorError::new_err(format!("KMS decryption failed: {e}")))?;

    // 验证密钥长度
    let expected_key_len = match algorithm {
        Algorithm::AES => 32,
        Algorithm::SM4 => 16,
    };
    if key.len() != expected_key_len {
        return Err(SafetensorError::new_err(format!(
            "Invalid key length for {}: expected {} bytes, got {}",
            envelope_meta.algorithm, expected_key_len, key.len()
        )));
    }

    // 读取 safetensors header size (8 字节, little-endian, 加密的)
    let mut size_buffer = [0u8; 8];
    file.read_exact(&mut size_buffer).map_err(|e| {
        SafetensorError::new_err(format!("Error reading size: {e}"))
    })?;

    // 解密 size（根据算法选择）
    decrypt_block(&key, &iv, &mut size_buffer, 0, &algorithm)?;

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

    // 解密 header（偏移 8 字节，即 size 字段之后）
    decrypt_block(&key, &iv, &mut header_buffer, 8, &algorithm)?;

    // 解析 safetensors metadata
    let metadata: Metadata = serde_json::from_slice(&header_buffer).map_err(|e| {
        SafetensorError::new_err(format!("Error deserializing header: {e}"))
    })?;

    let source = EncryptedSource {
        file: Mutex::new(file),
        key,
        iv,
        data_offset,
        algorithm,
    };

    Ok(Some((n, metadata, source)))
}

/// 使用指定算法解密数据块
fn decrypt_block(key: &[u8], iv: &[u8], buffer: &mut [u8], offset: u64, algorithm: &Algorithm) -> PyResult<()> {
    match algorithm {
        Algorithm::AES => {
            let mut cipher = AesCtr::new_from_slices(key, iv).map_err(|e| {
                SafetensorError::new_err(format!("Error creating AES cipher: {e}"))
            })?;
            if offset > 0 {
                cipher.try_seek(offset).map_err(|e| {
                    SafetensorError::new_err(format!("Error seeking AES cipher: {e}"))
                })?;
            }
            cipher.apply_keystream(buffer);
        }
        Algorithm::SM4 => {
            let mut cipher = Sm4Ctr::new_from_slices(key, iv).map_err(|e| {
                SafetensorError::new_err(format!("Error creating SM4 cipher: {e}"))
            })?;
            if offset > 0 {
                cipher.try_seek(offset).map_err(|e| {
                    SafetensorError::new_err(format!("Error seeking SM4 cipher: {e}"))
                })?;
            }
            cipher.apply_keystream(buffer);
        }
    }
    Ok(())
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

    decrypt_block(&source.key, &source.iv, &mut buffer, cipher_offset, &source.algorithm)?;

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

        for (start, stop) in &indices {
            let len = stop - start;
            // 文件中的实际位置
            let file_offset = (source.data_offset + tensor_start_offset + start) as u64;

            file.seek(SeekFrom::Start(file_offset)).map_err(|e| {
                SafetensorError::new_err(format!("Error seeking file: {e}"))
            })?;

            file.read_exact(&mut buffer[buffer_offset..buffer_offset + len]).map_err(|e| {
                SafetensorError::new_err(format!("Error reading file: {e}"))
            })?;

            buffer_offset += len;
        }
    }

    // 解密所有块
    buffer_offset = 0;
    for (start, stop) in indices {
        let len = stop - start;
        let cipher_offset = (tensor_start_offset + start) as u64;

        decrypt_block(
            &source.key,
            &source.iv,
            &mut buffer[buffer_offset..buffer_offset + len],
            cipher_offset,
            &source.algorithm
        )?;

        buffer_offset += len;
    }

    Ok(buffer)
}
