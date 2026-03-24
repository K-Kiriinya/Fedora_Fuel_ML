"""
Authentication Module for Fedora Fuel ML
-----------------------------------------

Provides JWT-based authentication for protected endpoints.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
import os

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token scheme
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Parameters
    ----------
    data : dict
        Data to encode in the token (e.g., {"sub": "username"})
    expires_delta : timedelta, optional
        Token expiration time

    Returns
    -------
    str
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """
    Decode and verify a JWT token.

    Parameters
    ----------
    token : str
        JWT token to decode

    Returns
    -------
    dict
        Decoded token payload

    Raises
    ------
    HTTPException
        If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Dependency to get the current authenticated user.

    Parameters
    ----------
    credentials : HTTPAuthorizationCredentials
        HTTP Authorization header with Bearer token

    Returns
    -------
    dict
        User information from token

    Raises
    ------
    HTTPException
        If authentication fails
    """
    token = credentials.credentials
    payload = decode_access_token(token)

    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {"username": username}


# Dummy user database (replace with real database in production)
# Note: Password will be hashed on first use (lazy initialization)
_users_initialized = False
fake_users_db = {}


def _initialize_users():
    """Initialize user database with hashed passwords (lazy initialization)."""
    global _users_initialized, fake_users_db
    if not _users_initialized:
        fake_users_db["admin"] = {
            "username": "admin",
            "full_name": "Admin User",
            "email": "admin@fedorafuel.ml",
            "hashed_password": get_password_hash("admin123"),  # Default password: admin123
            "disabled": False,
        }
        _users_initialized = True


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """
    Authenticate a user with username and password.

    Parameters
    ----------
    username : str
        Username
    password : str
        Plain text password

    Returns
    -------
    dict or None
        User dict if authentication succeeds, None otherwise
    """
    # Initialize users on first call
    _initialize_users()

    user = fake_users_db.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user
