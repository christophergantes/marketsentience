from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class Article(Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String,)
    description: Mapped[str] = mapped_column(String)
    url: Mapped[str] = mapped_column(String)
    sentiment: Mapped[str] = mapped_column(String)