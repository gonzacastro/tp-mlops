"""Esquemas de entrada y salida de la API.

`Auto` es una publicación de CarDekho tal como viene en el CSV, con las unidades adentro
del texto (`"23.4 kmpl"`, `"1248 CC"`). El modelo hace la limpieza, así que la API no
transforma nada: arma un DataFrame con estos campos y lo pasa al pipeline.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from car_price.config import ANIO_REFERENCIA, KM_MAX

EJEMPLO_AUTO = {
    "name": "Maruti Swift Dzire VDI",
    "year": 2014,
    "km_driven": 145500,
    "fuel": "Diesel",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner",
    "mileage": "23.4 kmpl",
    "engine": "1248 CC",
    "max_power": "74 bhp",
    "torque": "190Nm@ 2000rpm",
    "seats": 5,
}


class Auto(BaseModel):
    """Publicación de un auto usado, con los campos del CSV de CarDekho."""

    model_config = ConfigDict(json_schema_extra={"examples": [EJEMPLO_AUTO]})

    name: str = Field(
        description="Marca y modelo. La primera palabra se usa como marca.",
        examples=["Maruti Swift Dzire VDI", "Hyundai i20 Sportz 1.2"],
    )
    year: int = Field(
        ge=1980, le=ANIO_REFERENCIA,
        description=f"Año de fabricación. El dataset llega hasta {ANIO_REFERENCIA}.",
        examples=[2014],
    )
    km_driven: int = Field(
        ge=0, le=KM_MAX,
        description="Kilómetros recorridos.",
        examples=[145500],
    )
    fuel: Literal["Diesel", "Petrol", "CNG", "LPG"] = Field(
        description="Combustible.", examples=["Diesel"],
    )
    seller_type: Literal["Individual", "Dealer", "Trustmark Dealer"] = Field(
        description="Tipo de vendedor.", examples=["Individual"],
    )
    transmission: Literal["Manual", "Automatic"] = Field(
        description="Transmisión.", examples=["Manual"],
    )
    owner: Literal["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"] = Field(
        description="Cantidad de dueños anteriores.", examples=["First Owner"],
    )
    mileage: str | None = Field(
        default=None,
        description="Rendimiento con unidad: `kmpl` (nafta o diésel) o `km/kg` (GNC o GLP). "
                    "Si falta, el modelo lo imputa.",
        examples=["23.4 kmpl", "26.6 km/kg"],
    )
    engine: str | None = Field(
        default=None,
        description="Cilindrada en CC. Si falta, el modelo la imputa.",
        examples=["1248 CC"],
    )
    max_power: str | None = Field(
        default=None,
        description="Potencia máxima en bhp. Si falta, el modelo la imputa.",
        examples=["74 bhp"],
    )
    torque: str | None = Field(
        default=None,
        description="Torque como texto libre. El modelo no lo usa; se acepta para poder "
                    "mandar una fila del CSV tal cual.",
        examples=["190Nm@ 2000rpm"],
    )
    seats: float | None = Field(
        default=None, ge=2, le=14,
        description="Cantidad de asientos.",
        examples=[5],
    )


class Prediccion(BaseModel):
    """Precio estimado y versión del modelo que lo calculó."""

    model_config = ConfigDict(json_schema_extra={"examples": [
        {"precio_inr": 457722, "version_modelo": "2", "alias": "champion"},
    ]})

    precio_inr: int = Field(description="Precio estimado en rupias indias, redondeado.")
    version_modelo: str = Field(description="Versión de `car_price_gb` en el registry de MLflow.")
    alias: str = Field(description="Alias con el que se cargó el modelo.")


class MetricasTest(BaseModel):
    """Métricas del modelo sobre el test (1.339 autos que no vio al entrenar)."""

    rmse_log: float = Field(description="RMSE en escala log(precio).")
    r2_log: float = Field(description="R² en escala log(precio).")
    mae_inr: float = Field(description="Error absoluto medio en rupias.")


class InfoModelo(BaseModel):
    """Qué modelo está sirviendo la API."""

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "nombre": "car_price_gb", "version": "2", "alias": "champion",
        "run_id": "ded7bb3fbf18446b9d9910d438fc8386",
        "metricas_test": {"rmse_log": 0.2049, "r2_log": 0.9224, "mae_inr": 71844.0},
    }]})

    nombre: str = Field(description="Nombre del modelo en el registry de MLflow.")
    version: str = Field(description="Versión cargada.")
    alias: str = Field(description="Alias con el que se cargó.")
    run_id: str = Field(description="Run de MLflow que generó el modelo.")
    metricas_test: MetricasTest | None = Field(
        description="Métricas en test guardadas como tags de la versión.")


class Recarga(BaseModel):
    """Resultado de volver a cargar el champion."""

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "version_anterior": "2", "version_actual": "3", "cambio": True,
    }]})

    version_anterior: str | None = Field(description="Versión que se estaba sirviendo, o null.")
    version_actual: str = Field(description="Versión que se sirve ahora.")
    cambio: bool = Field(description="Si el champion cambió de versión.")


class Salud(BaseModel):
    """Estado del servicio."""

    status: Literal["ok"] = Field(description="Siempre `ok` si la API responde.")
    modelo_cargado: bool = Field(description="Si hay un champion cargado para predecir.")
