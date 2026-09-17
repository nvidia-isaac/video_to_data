"""Store cleanup reclaimed-byte accounting as a 64-bit integer."""

from alembic import op
import sqlalchemy as sa


revision = "0010_cleanup_bytes_bigint"
down_revision = "0009_intermediate_cleanup_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table("intermediate_cleanup_jobs") as batch:
            batch.alter_column(
                "reclaimed_bytes", existing_type=sa.Integer(),
                type_=sa.BigInteger(), existing_nullable=True,
            )
    else:
        op.alter_column(
            "intermediate_cleanup_jobs", "reclaimed_bytes",
            existing_type=sa.Integer(), type_=sa.BigInteger(),
            existing_nullable=True,
        )


def downgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table("intermediate_cleanup_jobs") as batch:
            batch.alter_column(
                "reclaimed_bytes", existing_type=sa.BigInteger(),
                type_=sa.Integer(), existing_nullable=True,
            )
    else:
        op.alter_column(
            "intermediate_cleanup_jobs", "reclaimed_bytes",
            existing_type=sa.BigInteger(), type_=sa.Integer(),
            existing_nullable=True,
        )
