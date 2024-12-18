from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

	DATA_DIR: str
	BRUT_DATA_PATH: str
	COPICK_VOX_SIZE: int
	TRAIN_TOMO_TYPE: str


CONF = Settings()

if __name__ == "__main__":
	print(CONF)
	import os
	print(CONF.DATA_DIR)
	print(os.path.exists(CONF.DATA_DIR))


