import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  // The lab is opened at 127.0.0.1 while the dev server binds 0.0.0.0.
  // Without this, Next blocks the HMR socket and the page never hydrates.
  allowedDevOrigins: ["127.0.0.1", "localhost"],
};

export default nextConfig;
