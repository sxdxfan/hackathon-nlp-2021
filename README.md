## Установка и запуск

Для работы сервиса необходимо установить `Docker` и `docker-compose`:
```bash
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) \
    stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker $(whoami)
sudo curl -L "https://github.com/docker/compose/releases/download/1.25.5/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

Для билда сервиса выполните команду:
```
sudo docker-compose build
```

Для запуска и перезапуска сервиса выполните команду:
```
sudo docker-compose up -d --force-recreate
```

Сервис поднимается по умолчанию на `5000` порту, загружаемые файлы хранятся в папке `uploads`, модели - в папке `models`.