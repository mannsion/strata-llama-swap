# llama-swap: one address for every model; starts the one a request names (docker compose) and stops the other.
FROM docker:28-cli
ARG LS_VER=260
RUN wget -qO- https://github.com/mostlygeek/llama-swap/releases/download/v${LS_VER}/llama-swap_${LS_VER}_linux_amd64.tar.gz \
    | tar xz -C /usr/local/bin llama-swap
ENTRYPOINT ["llama-swap"]
