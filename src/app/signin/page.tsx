import Link from "next/link";
import { SignInForm } from "@/components/auth/forms";

export default function SignInPage() {
  return (
    <main className="mx-auto flex min-h-screen max-w-md flex-col justify-center gap-5 px-6">
      <h1 className="text-3xl font-semibold">QED-Net sign in</h1>
      <p>Use the email address approved for this private research workspace.</p>
      <SignInForm />
      <div className="flex justify-between text-sm">
        <Link className="underline" href="/signup">Create account</Link>
        <Link className="underline" href="/forgot-password">Forgot password?</Link>
      </div>
      <p className="text-sm text-muted-foreground">Research and educational use only. Not for diagnosis or treatment.</p>
    </main>
  );
}
