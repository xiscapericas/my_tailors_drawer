# Planteatelo

Sistema de monitorización de plantas que combina sensores, cámara e inteligencia artificial para vigilar la salud de tus plantas y generar alertas y recomendaciones.

## Qué hace

1. Captura imágenes periódicas de la planta.
2. Lee métricas del sensor (humedad, temperatura, luz, conductividad).
3. Envía los datos a AWS.
4. Analiza el estado visual con IA *(previsto)*.
5. Genera alertas y recomendaciones *(previsto)*.

## Hardware

| Componente | Modelo |
|---|---|
| Orquestador | Raspberry Pi Zero 2 W |
| Sensor | Xiaomi Flower Care (BLE) |
| Cámara | Freenove 5MP |

## Arquitectura

```text
Raspberry Pi
  ├── Cámara → imagen
  ├── Sensor BLE → métricas
  └── MQTT (AWS IoT Core)
           │
           ▼
      Lambda (process_reading)
           │
           ├── DynamoDB (lecturas)
           └── S3 (imágenes)
```

**AWS (eu-west-1):** IoT Core, Lambda, DynamoDB, S3 — gestionado con Terraform.

## Estructura del repositorio

```text
planteatelo/
├── infra/terraform/     # Infraestructura AWS
├── backend/
│   ├── raspberry/       # Scripts en la Pi (click, sensor, status)
│   └── lambdas/         # Funciones Lambda
├── frontend/            # App web (pendiente)
└── docs/
```

## Estado actual

| Fase | Estado |
|---|---|
| Captura local (foto + sensor + JSON) | Operativo |
| Infraestructura AWS | Desplegada |
| Conexión Pi → IoT Core | Pendiente |
| Upload de imágenes a S3 | Pendiente |
| Análisis con GPT Vision | Pendiente |

## Despliegue de infraestructura

```bash
cd infra/terraform
AWS_PROFILE=planteatelo terraform plan
AWS_PROFILE=planteatelo terraform apply
```

Recursos principales:

- **S3:** `s3://planteatelo/planteatelo-dev-images/`
- **DynamoDB:** `planteatelo-dev-readings`, `planteatelo-dev-analyses`
- **IoT topic:** `planteatelo/plant_001/readings`
- **Lambda:** `planteatelo-dev-process-reading`

## Próximos pasos

- Certificados IoT y publicación MQTT desde la Raspberry.
- Subida de imágenes a S3.
- Lambda de análisis con OpenAI Vision.
