import Link from "next/link";
import { ForgotPasswordForm } from "@/components/auth/forms";

export default function ForgotPasswordPage() {
  return (
    <main className="mx-auto flex min-h-screen max-w-md flex-col justify-center gap-5 px-6">
      <h1 className="text-3xl font-semibold">Reset your password</h1>
      <p>We’ll send a reset link if this address is enabled for QED-Net.</p>
      <ForgotPasswordForm />
      <Link className="text-sm underline" href="/signin">Back to sign in</Link>
    </main>
  );
}
