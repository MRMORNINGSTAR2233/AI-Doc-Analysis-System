FROM node:18-alpine

WORKDIR /app

# Install dependencies first for better caching
COPY package.json yarn.lock ./
RUN yarn install --frozen-lockfile

# Copy the rest of the application
COPY . .

# Build the application
RUN yarn build

# Expose the port
EXPOSE 3000

# Run the application
CMD ["yarn", "start"] 