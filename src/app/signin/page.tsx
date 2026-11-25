export default function SignInPage() {
  return (
    <main className="mx-auto flex min-h-screen max-w-xl flex-col justify-center gap-5 px-6">
      <h1 className="text-3xl font-semibold">QED-Net research sign in</h1>
      <p>Sign in with an account authorized by your organization.</p>
      <a
        className="w-fit rounded-md bg-primary px-4 py-2 text-primary-foreground"
        href="/api/auth/signin/organization"
      >
        Continue with organization account
      </a>
      <p className="text-sm text-muted-foreground">
        Research and educational use only. Not for diagnosis or treatment.
      </p>
    </main>
  );
}
