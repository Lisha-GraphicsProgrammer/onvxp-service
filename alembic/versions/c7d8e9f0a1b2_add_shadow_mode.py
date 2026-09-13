"""add shadow mode support

Revision ID: c7d8e9f0a1b2
Revises: a1b2c3d4e5f6
Create Date: 2026-09-13
"""
from alembic import op
import sqlalchemy as sa

revision = 'c7d8e9f0a1b2'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'incidents',
        sa.Column('is_shadow', sa.Boolean(), nullable=False, server_default=sa.false())
    )


def downgrade():
    op.drop_column('incidents', 'is_shadow')