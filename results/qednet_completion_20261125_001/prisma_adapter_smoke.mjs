const { db } = await import("../../src/lib/db.ts");
const email = `qednet-smoke-${Date.now()}@example.invalid`;
try {
  const created = await db.user.create({ data: { email } });
  const found = await db.user.findUnique({ where: { email } });
  const deleted = await db.user.delete({ where: { email } });
  console.log(`CRUD_OK=${created.email === found?.email && deleted.email === email}`);
} finally {
  await db.$disconnect();
}
