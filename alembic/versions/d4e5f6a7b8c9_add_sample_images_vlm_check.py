"""add sample_images and vlm_check to training_jobs

Revision ID: d4e5f6a7b8c9
Revises: 0fffe2910e65
Create Date: 2026-09-19
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'd4e5f6a7b8c9'
down_revision = '0fffe2910e65'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('training_jobs', sa.Column('sample_images', postgresql.JSONB(), nullable=True))
    op.add_column('training_jobs', sa.Column('vlm_check', postgresql.JSONB(), nullable=True))


def downgrade():
    op.drop_column('training_jobs', 'vlm_check')
    op.drop_column('training_jobs', 'sample_images')