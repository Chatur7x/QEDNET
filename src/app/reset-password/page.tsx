import Link from "next/link";
import { ResetPasswordForm } from "@/components/auth/forms";

export default function ResetPasswordPage() {
  return (
    <main className="mx-auto flex min-h-screen max-w-md flex-col justify-center gap-5 px-6">
      <h1 className="text-3xl font-semibold">Choose a new password</h1>
      <ResetPasswordForm />
      <Link className="text-sm underline" href="/signin">Back to sign in</Link>
    </main>
  );
}
