# syntax=docker/dockerfile:1
FROM python:3.13-slim AS build
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-cache-dir --wheel-dir /wheels ".[train]"

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ONCOAI_HOST=0.0.0.0 ONCOAI_PORT=8080
RUN useradd --system --uid 10001 --no-create-home oncoai
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
USER 10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz').status==200 else 1)"
ENTRYPOINT ["oncoai"]
CMD ["serve"]
