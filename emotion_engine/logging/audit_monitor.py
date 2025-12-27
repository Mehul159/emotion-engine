"""
Audit and monitoring utilities for Emotion Engine.
"""
import logging
from datetime import datetime

def audit_event(event_type: str, details: dict):
    logger = logging.getLogger("audit")
    logger.info(f"AUDIT | {datetime.utcnow().isoformat()} | {event_type} | {details}")

def monitor_metric(metric: str, value: float, tags: dict = None):
    logger = logging.getLogger("monitor")
    tag_str = f" | {tags}" if tags else ""
    logger.info(f"METRIC | {datetime.utcnow().isoformat()} | {metric} | {value}{tag_str}")
