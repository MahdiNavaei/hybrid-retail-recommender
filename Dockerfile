# EN: Use a lightweight Python base image
# FA: استفاده از ایمیج پایتون سبک به عنوان پایه
FROM python:3.11-slim

# EN: Set workdir inside the container
# FA: تنظیم پوشه کاری در کانتینر
WORKDIR /app

# EN: Copy requirements first to leverage Docker layer caching
# FA: ابتدا فایل وابستگی‌ها را کپی می‌کنیم تا لایه‌های داکر کش شوند
COPY requirements.txt .

# EN: Install dependencies without cache for smaller image
# FA: نصب وابستگی‌ها بدون کش برای کاهش حجم ایمیج
RUN pip install --no-cache-dir -r requirements.txt

# EN: Copy the rest of the project
# FA: کپی بقیه پروژه
COPY . .

# EN: Expose API port
# FA: پورت API را باز می‌کنیم
EXPOSE 8000

# EN: Default command to start FastAPI with uvicorn
# FA: فرمان پیش‌فرض برای اجرای FastAPI با uvicorn
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
