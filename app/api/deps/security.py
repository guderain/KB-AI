import time

from fastapi import Header, HTTPException, Request, status

from app.core.config import get_settings
from app.services.dependencies import get_redis_client


def _extract_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get('x-forwarded-for', '')
    if forwarded_for:
        return forwarded_for.split(',')[0].strip()
    client = request.client
    return client.host if client else 'unknown'


def _verify_api_key(x_api_key: str | None) -> None:
    settings = get_settings()
    keys = settings.api_keys_list
    if not keys:
        return
    if not x_api_key or x_api_key not in keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='invalid api key',
        )


def _rate_limit(scope: str, client_ip: str, limit_per_minute: int) -> None:
    if limit_per_minute <= 0:
        return
    now_minute = int(time.time() // 60)
    redis_client = get_redis_client()
    key = f'rate:{scope}:{client_ip}:{now_minute}'
    try:
        count = redis_client.incr(key)
        if count == 1:
            redis_client.expire(key, 120)
        if count > limit_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f'rate limit exceeded: {limit_per_minute}/min',
            )
    except HTTPException:
        raise
    except Exception:
        return


def guard_read_write(
    request: Request,
    x_api_key: str | None = Header(default=None, alias='X-API-Key'),
) -> None:
    settings = get_settings()
    _verify_api_key(x_api_key)
    _rate_limit(
        scope='api',
        client_ip=_extract_client_ip(request),
        limit_per_minute=settings.RATE_LIMIT_PER_MINUTE,
    )


def guard_ingest(
    request: Request,
    x_api_key: str | None = Header(default=None, alias='X-API-Key'),
) -> None:
    settings = get_settings()
    _verify_api_key(x_api_key)
    _rate_limit(
        scope='ingest',
        client_ip=_extract_client_ip(request),
        limit_per_minute=settings.INGEST_RATE_LIMIT_PER_MINUTE,
    )
