# SHARP dashboard: build the Vite SPA, serve it via nginx and proxy /predict.
FROM node:22-alpine AS build

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
# Serve from the root inside the container (GitHub Pages uses VITE_BASE=/Projet-ia/).
ENV VITE_BASE=/
RUN npm run build

FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
