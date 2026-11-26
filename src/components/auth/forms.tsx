"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { createSupabaseBrowserClient } from "@/lib/supabase/browser";

function localPath(value: string | null) {
  return value?.startsWith("/") && !value.startsWith("//") ? value : "/";
}

function FormMessage({ children, error = false }: { children: string; error?: boolean }) {
  return <p role={error ? "alert" : "status"} className={error ? "text-sm text-red-700" : "text-sm text-emerald-800"}>{children}</p>;
}

export function SignInForm({ initialError }: { initialError?: string }) {
  const router = useRouter();
  const [message, setMessage] = useState("");
  const pageError = initialError ?? "";
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setMessage("");
    const form = new FormData(event.currentTarget);
    const supabase = createSupabaseBrowserClient();
    const { error } = await supabase.auth.signInWithPassword({
      email: String(form.get("email") ?? "").trim(),
      password: String(form.get("password") ?? ""),
    });
    setBusy(false);
    if (error) {
      setMessage(/email not confirmed/i.test(error.message)
        ? "This account's email is not confirmed yet. Open the latest confirmation email, or use Create account to request another link."
        : "Sign-in failed. Check your email and password, then try again.");
      return;
    }
    const next = new URLSearchParams(window.location.search).get("callbackUrl");
    router.replace(localPath(next));
    router.refresh();
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <label className="block space-y-1 text-sm">Email<input className="w-full rounded border px-3 py-2" name="email" type="email" autoComplete="email" required /></label>
      <label className="block space-y-1 text-sm">Password<input className="w-full rounded border px-3 py-2" name="password" type="password" autoComplete="current-password" required /></label>
      <button className="rounded bg-primary px-4 py-2 text-primary-foreground" type="submit" disabled={busy}>{busy ? "Signing in…" : "Sign in"}</button>
      {pageError && <FormMessage error>{pageError}</FormMessage>}
      {message && <FormMessage error>{message}</FormMessage>}
    </form>
  );
}

export function SignUpForm() {
  const router = useRouter();
  const [message, setMessage] = useState("");
  const [error, setError] = useState(false);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setMessage("");
    const form = new FormData(event.currentTarget);
    try {
      const response = await fetch("/api/auth/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: form.get("email") }),
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) {
        setError(true);
        setMessage(result.error || "Sign-up failed. Try again.");
        return;
      }
      const supabase = createSupabaseBrowserClient();
      const { data: { session }, error: signUpError } = await supabase.auth.signUp({
        email: String(form.get("email") ?? "").trim(),
        password: String(form.get("password") ?? ""),
        options: {
          data: { username: String(form.get("username") ?? "").trim() },
          emailRedirectTo: `${window.location.origin}/auth/confirm?next=%2F`,
        },
      });
      if (!signUpError && session) {
        router.replace("/");
        router.refresh();
        return;
      }
      setError(true);
      setMessage("Account creation did not start a session. Try signing in or use Forgot password if you already have an account.");
    } catch {
      setError(true);
      setMessage("Sign-up could not reach the authentication service. Check your connection and try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <label className="block space-y-1 text-sm">Username<input className="w-full rounded border px-3 py-2" name="username" type="text" autoComplete="username" maxLength={64} required /></label>
      <label className="block space-y-1 text-sm">Email<input className="w-full rounded border px-3 py-2" name="email" type="email" autoComplete="email" required /></label>
      <label className="block space-y-1 text-sm">Password<input className="w-full rounded border px-3 py-2" name="password" type="password" autoComplete="new-password" minLength={12} maxLength={128} required /></label>
      <button className="rounded bg-primary px-4 py-2 text-primary-foreground" type="submit" disabled={busy}>{busy ? "Creating account…" : "Create account"}</button>
      {message && <FormMessage error={error}>{message}</FormMessage>}
    </form>
  );
}

export function ForgotPasswordForm() {
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    const email = String(new FormData(event.currentTarget).get("email") ?? "").trim();
    const supabase = createSupabaseBrowserClient();
    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}/auth/confirm?next=%2Freset-password`,
    });
    setBusy(false);
    setMessage(error ? "Could not send a reset email. Try again later." : "If the address is enabled, a reset link will be sent.");
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <label className="block space-y-1 text-sm">Email<input className="w-full rounded border px-3 py-2" name="email" type="email" autoComplete="email" required /></label>
      <button className="rounded bg-primary px-4 py-2 text-primary-foreground" type="submit" disabled={busy}>{busy ? "Sending…" : "Send reset link"}</button>
      {message && <FormMessage>{message}</FormMessage>}
    </form>
  );
}

export function ResetPasswordForm() {
  const router = useRouter();
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    const password = String(new FormData(event.currentTarget).get("password") ?? "");
    const supabase = createSupabaseBrowserClient();
    const { error } = await supabase.auth.updateUser({ password });
    setBusy(false);
    if (error) {
      setMessage("Password update failed. Request a fresh reset link and try again.");
      return;
    }
    router.replace("/");
    router.refresh();
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <label className="block space-y-1 text-sm">New password<input className="w-full rounded border px-3 py-2" name="password" type="password" autoComplete="new-password" minLength={12} maxLength={128} required /></label>
      <button className="rounded bg-primary px-4 py-2 text-primary-foreground" type="submit" disabled={busy}>{busy ? "Updating…" : "Update password"}</button>
      {message && <FormMessage error>{message}</FormMessage>}
    </form>
  );
}

export function SignOutButton() {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  async function signOut() {
    setBusy(true);
    await createSupabaseBrowserClient().auth.signOut();
    router.replace("/signin");
    router.refresh();
  }
  return <button type="button" onClick={signOut} disabled={busy} className="rounded border border-stone-300 px-3 py-1.5 text-sm text-stone-700 hover:bg-stone-100">{busy ? "Signing out…" : "Sign out"}</button>;
}
