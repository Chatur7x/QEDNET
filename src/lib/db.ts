import { PrismaBetterSqlite3 } from '@prisma/adapter-better-sqlite3'
import { PrismaClient } from '@prisma/client'

const databaseUrl = process.env.QEDNET_PRISMA_DATABASE_URL
if (!databaseUrl?.startsWith('file:')) {
  throw new Error('QEDNET_PRISMA_DATABASE_URL must be a file: SQLite URL')
}

const adapter = new PrismaBetterSqlite3({ url: databaseUrl })

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined
}

export const db =
  globalForPrisma.prisma ??
  new PrismaClient({
    adapter,
    log: process.env.NODE_ENV === 'development' ? ['query', 'error'] : ['error'],
  })

if (process.env.NODE_ENV !== 'production') globalForPrisma.prisma = db
