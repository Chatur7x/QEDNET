import type { Metadata } from "next";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

export const metadata: Metadata = {
  title: "QED-Net 2.0 — Hybrid Quantum–Classical ML Research Platform",
  description:
    "Research dashboard for QED-Net 2.0: hybrid quantum–classical machine learning for biomedical disease-risk research. Benchmarking, explainability, uncertainty and quantum-advantage certificates from real experiments. Research and educational use only — not a medical device.",
  keywords: [
    "quantum machine learning",
    "hybrid quantum-classical",
    "QSVM",
    "VQC",
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
