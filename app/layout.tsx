import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans_JP, Shippori_Mincho } from "next/font/google";
import "./globals.css";
import { TooltipProvider } from "@/components/ui/tooltip";

const sans = IBM_Plex_Sans_JP({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-sans-jp",
  display: "swap",
});

const display = Shippori_Mincho({
  subsets: ["latin"],
  weight: ["500", "600"],
  variable: "--font-display-jp",
  display: "swap",
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono-jp",
  display: "swap",
});

export const metadata: Metadata = {
  title: "ピッチデザイン・ラボ",
  description:
    "握り、指の力、腕の振りから変化球の軌道を設計する。抗力とマグヌス力で本塁までの球筋を計算します。",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="ja" className={`dark h-full antialiased ${sans.variable} ${display.variable} ${mono.variable}`}>
      <body className="min-h-full">
        <TooltipProvider>{children}</TooltipProvider>
      </body>
    </html>
  );
}
