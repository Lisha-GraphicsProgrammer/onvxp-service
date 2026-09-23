from sqlalchemy import Column, Integer, Text, ForeignKey, Boolean
from sqlalchemy import TIMESTAMP
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Site(Base):
    __tablename__ = "sites"
    id         = Column(Integer, primary_key=True)
    name       = Column(Text, nullable=False)
    location   = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())


class User(Base):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True)
    email         = Column(Text, unique=True, nullable=False)
    password_hash = Column(Text, nullable=False)
    name          = Column(Text)
    role          = Column(Text, default="admin", nullable=False)
    site_id       = Column(Integer, ForeignKey("sites.id"))
    created_at    = Column(TIMESTAMP(timezone=True), server_default=func.now())
    last_login    = Column(TIMESTAMP(timezone=True))


class Zone(Base):
    __tablename__ = "zones"
    id         = Column(Integer, primary_key=True)
    site_id    = Column(Integer, ForeignKey("sites.id", ondelete="CASCADE"), nullable=False)
    name       = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())


class Camera(Base):
    __tablename__ = "cameras"
    id         = Column(Integer, primary_key=True)
    site_id    = Column(Integer, ForeignKey("sites.id", ondelete="CASCADE"), nullable=False)
    zone_id    = Column(Integer, ForeignKey("zones.id", ondelete="SET NULL"), nullable=True)
    name       = Column(Text, nullable=False)
    location   = Column(Text)
    source     = Column(Text, nullable=False)
    status     = Column(Text, default="offline", nullable=False)
    fps        = Column(Integer)
    resolution = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())


class Rule(Base):
    __tablename__ = "rules"
    id          = Column(Integer, primary_key=True)
    site_id     = Column(Integer, ForeignKey("sites.id", ondelete="CASCADE"), nullable=False)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    instruction = Column(Text, nullable=False)
    config_json = Column(JSONB, nullable=False)
    pipeline_id = Column(Text)
    # Valid values: "active", "inactive", "pending_training", "replaced", "deleted"
    status      = Column(Text, default="active", nullable=False)
    severity    = Column(Text)
    created_at  = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at  = Column(TIMESTAMP(timezone=True), server_default=func.now())


class RuleCamera(Base):
    __tablename__ = "rule_cameras"
    id        = Column(Integer, primary_key=True)
    rule_id   = Column(Integer, ForeignKey("rules.id", ondelete="CASCADE"), nullable=False)
    camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)


class Incident(Base):
    __tablename__ = "incidents"
    id               = Column(Integer, primary_key=True)
    rule_id          = Column(Integer, ForeignKey("rules.id", ondelete="CASCADE"), nullable=False)
    camera_id        = Column(Integer, ForeignKey("cameras.id"))
    site_id          = Column(Integer, ForeignKey("sites.id", ondelete="CASCADE"), nullable=False)
    timestamp        = Column(TIMESTAMP(timezone=True), server_default=func.now())
    frame_number     = Column(Integer)
    person_track_id  = Column(Integer)
    violation_type   = Column(Text)
    detected_objects = Column(JSONB)
    missing_gear     = Column(JSONB)
    zone             = Column(Text)
    bbox             = Column(JSONB)
    screenshot_path  = Column(Text)
    clip_path        = Column(Text)
    severity         = Column(Text)
    alert_message    = Column(Text)
    reviewed         = Column(Boolean, default=False, nullable=False)
    review_status    = Column(Text)
    reviewed_by      = Column(Integer, ForeignKey("users.id"))
    reviewed_at      = Column(TIMESTAMP(timezone=True))


class Setting(Base):
    __tablename__ = "settings"
    id         = Column(Integer, primary_key=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    site_id    = Column(Integer, ForeignKey("sites.id", ondelete="CASCADE"))
    key        = Column(Text, nullable=False)
    value      = Column(JSONB)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())


class TrainingJob(Base):
    __tablename__ = "training_jobs"
    id              = Column(Integer, primary_key=True)
    site_id         = Column(Integer, nullable=False)
    class_name      = Column(Text, nullable=False)
    rule_id         = Column(Integer, ForeignKey("rules.id", ondelete="SET NULL"), nullable=True)
    status          = Column(Text, default="pending", nullable=False)
    current_stage   = Column(Text)
    stages          = Column(JSONB, default=list, nullable=False)
    dataset_info    = Column(JSONB)
    checkpoint_path = Column(Text)
    model_path      = Column(Text)
    metrics         = Column(JSONB)
    sample_images   = Column(JSONB)
    vlm_check       = Column(JSONB)
    error           = Column(Text)
    created_at      = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at      = Column(TIMESTAMP(timezone=True), server_default=func.now())