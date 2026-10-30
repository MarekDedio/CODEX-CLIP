from __future__ import annotations

import ctypes
from ctypes import wintypes
import os


GROQ_CREDENTIAL_TARGET = "SpeechCutter/GroqApiKey"
GEMINI_CREDENTIAL_TARGET = "SpeechCutter/GeminiApiKey"
GROQ_CORRECTION_CREDENTIAL_TARGET = "SpeechCutter/GroqCorrectionApiKey"
GROQ_CORRECTION_CREDENTIAL_TARGETS = (
    GROQ_CORRECTION_CREDENTIAL_TARGET,
    "SpeechCutter/GroqCorrectionApiKey2",
    "SpeechCutter/GroqCorrectionApiKey3",
    "SpeechCutter/GroqCorrectionApiKey4",
    "SpeechCutter/GroqCorrectionApiKey5",
    "SpeechCutter/GroqCorrectionApiKey6",
)

_CRED_TYPE_GENERIC = 1
_CRED_PERSIST_LOCAL_MACHINE = 2
_ERROR_NOT_FOUND = 1168


class _CREDENTIALW(ctypes.Structure):
    pass


_PCREDENTIALW = ctypes.POINTER(_CREDENTIALW)
_CREDENTIALW._fields_ = [
    ("Flags", wintypes.DWORD),
    ("Type", wintypes.DWORD),
    ("TargetName", wintypes.LPWSTR),
    ("Comment", wintypes.LPWSTR),
    ("LastWritten", wintypes.FILETIME),
    ("CredentialBlobSize", wintypes.DWORD),
    ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
    ("Persist", wintypes.DWORD),
    ("AttributeCount", wintypes.DWORD),
    ("Attributes", ctypes.c_void_p),
    ("TargetAlias", wintypes.LPWSTR),
    ("UserName", wintypes.LPWSTR),
]


def load_api_key(target: str) -> str | None:
    if os.name != "nt":
        return None
    advapi32 = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    cred_read = advapi32.CredReadW
    cred_read.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(_PCREDENTIALW)]
    cred_read.restype = wintypes.BOOL
    cred_free = advapi32.CredFree
    cred_free.argtypes = [ctypes.c_void_p]

    credential_pointer = _PCREDENTIALW()
    if not cred_read(target, _CRED_TYPE_GENERIC, 0, ctypes.byref(credential_pointer)):
        return None
    try:
        credential = credential_pointer.contents
        if not credential.CredentialBlob or not credential.CredentialBlobSize:
            return None
        blob = ctypes.string_at(credential.CredentialBlob, credential.CredentialBlobSize)
        return blob.decode("utf-16-le")
    except (UnicodeDecodeError, ValueError):
        return None
    finally:
        cred_free(credential_pointer)


def save_api_key(target: str, api_key: str) -> bool:
    if os.name != "nt" or not api_key:
        return False
    advapi32 = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    cred_write = advapi32.CredWriteW
    cred_write.argtypes = [ctypes.POINTER(_CREDENTIALW), wintypes.DWORD]
    cred_write.restype = wintypes.BOOL

    blob = api_key.encode("utf-16-le")
    blob_buffer = ctypes.create_string_buffer(blob)
    credential = _CREDENTIALW()
    credential.Type = _CRED_TYPE_GENERIC
    credential.TargetName = target
    credential.CredentialBlobSize = len(blob)
    credential.CredentialBlob = ctypes.cast(blob_buffer, ctypes.POINTER(ctypes.c_ubyte))
    credential.Persist = _CRED_PERSIST_LOCAL_MACHINE
    credential.UserName = "Speech Cutter"
    return bool(cred_write(ctypes.byref(credential), 0))


def delete_api_key(target: str) -> bool:
    if os.name != "nt":
        return False
    advapi32 = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    cred_delete = advapi32.CredDeleteW
    cred_delete.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    cred_delete.restype = wintypes.BOOL
    if cred_delete(target, _CRED_TYPE_GENERIC, 0):
        return True
    return ctypes.get_last_error() == _ERROR_NOT_FOUND
