import type { Metadata } from "next";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

export const metadata: Metadata = {
  title: "QED-Net Research — Whole-Body Health Research Atlas",
  description:
    "Research workspace for transparent evaluation of classical and simulator-based quantum ML on biomedical datasets, with a whole-body and women’s-health research atlas. Research and education only; not for diagnosis or clinical decisions.",
  keywords: [
    "quantum machine learning",
    "hybrid quantum-classical",
    "QSVM",
    "VQC",
    "whole-body health research",
    "women's health research",
    "biomedical data analytics",
    "explainable AI",
    "research platform",
  ],
  authors: [{ name: "QED-Net 2.0 Research Project" }],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased bg-background text-foreground">
        {children}
        <Toaster />
      </body>
    </html>
  );
}
