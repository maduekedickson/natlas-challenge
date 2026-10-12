"""SQLAlchemy ORM models for report persistence.

Stock items are normalized into their own table (rather than left inside the
structured_data JSON blob) specifically so trend queries like "ORS quantity
at Asaba PHC over time" are plain SQL, not JSON-parsing per request.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    facility_name: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    reporter_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    report_text: Mapped[str] = mapped_column(Text, nullable=False)
    structured_data: Mapped[str] = mapped_column(Text, nullable=False)
    services_available: Mapped[str | None] = mapped_column(Text, nullable=True)
    facility_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    adapter_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="text")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="success")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    stock_observations: Mapped[list["StockObservation"]] = relationship(
        back_populates="report", cascade="all, delete-orphan"
    )


class StockObservation(Base):
    __tablename__ = "stock_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("reports.id"), nullable=False, index=True)
    item: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    quantity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    unit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    report: Mapped["Report"] = relationship(back_populates="stock_observations")
