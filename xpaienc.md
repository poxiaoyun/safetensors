# Safetensors XpaiEnc 集成文档

本文档总结了 `safetensors` 为集成 `xpai_enc` 实现透明模型解密所做的修改。

## 概述

该集成允许 `safetensors` 在不将解密内容暴露到磁盘的情况下，实时解密加密的模型文件。它支持基于文件的配置（通过 `SAFETENSORS_ENC_CONFIG` 环境变量）和基于内存的配置（通过 `set_decryption_config` API）。

## 修改内容

### 1. `bindings/python/src/lib.rs`

*   **模块集成**: 添加了 `mod decryption;` 以包含解密逻辑。
*   **Storage 枚举**: 扩展了 `Storage` 枚举，增加了 `Encrypted(EncryptedSource)` 变体以处理加密文件源。
*   **Open::new**: 修改了初始化逻辑，使用 `decryption::create_encrypted_source` 检查加密配置。如果配置存在，则将存储初始化为 `Storage::Encrypted`。
*   **Open::get_tensor**: 增加了对 `Storage::Encrypted` 的处理。它计算请求张量的字节范围，并调用 `decryption::decrypt_buffer` 进行实时解密。
*   **PySafeSlice::__getitem__**:
    *   **重构切片逻辑**: 将切片提取逻辑重构为辅助函数 `extract_slices`，以支持 `transformers` 库所需的 `Ellipsis` (`...`) 语法。
    *   **加密支持**: 增加了对 `Storage::Encrypted` 的处理。它使用 `calculate_slice_indices` 计算请求切片的精确字节范围，通过 `decryption::decrypt_slices` 仅解密必要的数据块，并重构张量。
*   **新 API**: 添加了 `set_decryption_config(config_json: str)`，允许直接从 Python 字符串设置加密配置，从而实现更安全的无文件配置。

### 2. `bindings/python/src/decryption.rs` (新文件)

该文件封装了所有解密逻辑，以保持主 `lib.rs` 文件的整洁。

*   **结构体定义**: 定义了 `EncryptedFile`, `EncryptedSource`, 和 `EncryptionConfig` 用于处理配置和状态。
*   **全局配置**: 管理一个全局配置状态 (`GLOBAL_CONFIG`) 用于内存中的设置。
*   **create_encrypted_source**: 检查环境变量或全局配置，以确定文件是否应被视为加密文件。如果匹配，则初始化 `Aes256Ctr` 密码器。
*   **decrypt_buffer**: 解密文件中连续的字节范围。
*   **decrypt_slices**: 高效解密多个不连续的字节范围（用于切片操作）。

### 3. `bindings/python/py_src/safetensors/__init__.py`

*   将 `set_decryption_config` 函数导出到顶层 `safetensors` 包中。

## 使用方法

### 环境变量模式

将 `SAFETENSORS_ENC_CONFIG` 设置为包含加密模型密钥和 IV 的 JSON 配置文件路径。

```bash
export SAFETENSORS_ENC_CONFIG=/path/to/safetensors_config.json
```

### 内存模式 (更安全)

使用 `set_decryption_config` API 直接传递配置 JSON 字符串。

```python
import safetensors
import json

config = {
    "files": [
        {
            "file": "model.safetensors",
            "key": "hex_encoded_key",
            "iv": "hex_encoded_iv"
        }
    ]
}
safetensors.set_decryption_config(json.dumps(config))
```

## 维护说明

*   **Rebase 代码**: `lib.rs` 中的更改非常小，主要涉及为 `Storage::Encrypted` 添加匹配分支。核心逻辑已隔离在 `decryption.rs` 中。在 Rebase 时，请确保保留 `mod decryption;` 行和 `Storage` 枚举的更改。
*   **Ellipsis 支持**: `lib.rs` 中的 `extract_slices` 辅助函数提高了与 `transformers` 的兼容性，即使加密逻辑发生变化，也应保留该函数。
