export default function SetupRequiredPage() {
  return (
    <main className="mx-auto flex min-h-screen max-w-2xl flex-col justify-center gap-4 px-6">
      <h1 className="text-3xl font-semibold">Organization sign in is not configured</h1>
      <p>
        The research workspace is locked until its organization identity provider is configured.
        Ask the service administrator to set the required values in the hosting secret store.
      </p>
      <ul className="list-inside list-disc text-sm">
        <li>NEXTAUTH_URL and NEXTAUTH_SECRET</li>
        <li>OIDC_ISSUER, OIDC_CLIENT_ID, and OIDC_CLIENT_SECRET</li>
        <li>OIDC_ALLOWED_GROUP (and optionally OIDC_GROUP_CLAIM; defaults to groups)</li>
      </ul>
      <p className="text-sm text-muted-foreground">
        This site is a research platform, not a clinical decision-support service.
      </p>
    </main>
  );
}
