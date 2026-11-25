import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  reactStrictMode: true,
  async rewrites() {
    return [{ source: "/v2-api/:path*", destination: "http://127.0.0.1:8000/:path*" }];
  },
};

export default nextConfig;
