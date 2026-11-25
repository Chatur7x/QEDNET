export default function SetupRequiredPage() {
  return (
    <main className="mx-auto flex min-h-screen max-w-2xl flex-col justify-center gap-4 px-6">
      <h1 className="text-3xl font-semibold">Email sign-in is not configured</h1>
      <p>
        The research workspace is locked until its authentication project is configured.
        Ask the service administrator to configure the Supabase project and app environment settings.
      </p>
      <ul className="list-inside list-disc text-sm">
        <li>NEXT_PUBLIC_SITE_URL and QEDNET_ALLOWED_EMAIL</li>
        <li>NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY</li>
      </ul>
      <p className="text-sm text-muted-foreground">
        The account is restricted to one allowlisted email because user data is not tenant-isolated. This is a research platform, not a clinical decision-support service.
      </p>
    </main>
  );
}
