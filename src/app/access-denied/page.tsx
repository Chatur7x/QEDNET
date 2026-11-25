import Link from "next/link";

export default function AccessDeniedPage() {
  return (
    <main className="mx-auto flex min-h-screen max-w-xl flex-col justify-center gap-4 px-6">
      <h1 className="text-3xl font-semibold">This account is not enabled</h1>
      <p>QED-Net is restricted to the single email address configured for this research workspace.</p>
      <Link className="w-fit underline" href="/signin">Return to sign in</Link>
    </main>
  );
}
