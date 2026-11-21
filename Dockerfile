FROM python:3.13-slim
RUN pip install --no-cache-dir numpy==2.2.6
WORKDIR /bot
COPY bot.py nav.py mechfight.py /bot/
ENV PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
