import { defineConfig } from 'prisma/config'

const databaseUrl = process.env.QEDNET_PRISMA_DATABASE_URL

export default defineConfig({
  schema: 'prisma/schema.prisma',
  ...(databaseUrl ? { datasource: { url: databaseUrl } } : {}),
})
