import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "RetinaVessel AI | Research Segmentation",
  description: "Research-only retinal vessel segmentation workspace.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
