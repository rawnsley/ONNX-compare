# Pinned slim Debian + Node 24. node_modules and run.mjs are baked into the
# image; shared/ and cache/ are bind-mounted at runtime so the fixture and
# downloaded model are shared with the host.
FROM node:24-bookworm-slim
WORKDIR /onnx-compare/linux-x64
COPY package.json .
RUN npm install --silent
COPY run.mjs .
ENTRYPOINT ["node", "/onnx-compare/linux-x64/run.mjs"]
