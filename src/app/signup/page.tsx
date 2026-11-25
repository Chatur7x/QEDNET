import Link from "next/link";
import { SignUpForm } from "@/components/auth/forms";

export default function SignUpPage() {
  return (
    <main className="mx-auto flex min-h-screen max-w-md flex-col justify-center gap-5 px-6">
      <h1 className="text-3xl font-semibold">Create your QED-Net account</h1>
      <p>Only the email address enabled by the workspace administrator can register.</p>
      <SignUpForm />
      <Link className="text-sm underline" href="/signin">Back to sign in</Link>
      <p className="text-sm text-muted-foreground">Research and educational use only. Not for diagnosis or treatment.</p>
    </main>
  );
}
