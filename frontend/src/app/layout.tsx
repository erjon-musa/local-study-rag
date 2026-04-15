import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Study Notes",
  description:
    "Chat with your course notes using RAG + Gemma4. Built for finals prep.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
