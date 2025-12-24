FROM continuumio/miniconda3:latest

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml && \
    conda clean -afy

SHELL ["conda", "run", "-n", "georef-app", "/bin/bash", "-c"]

COPY . .

# Создаем необходимые директории с правильными правами
RUN mkdir -p temp static && \
    chmod 777 temp

EXPOSE 8000

CMD ["conda", "run", "--no-capture-output", "-n", "georef-app", \
     "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]