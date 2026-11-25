import type { NextAuthOptions } from "next-auth";
import { isAllowedOrganizationProfile } from "@/lib/auth-policy";

const issuer = process.env.OIDC_ISSUER?.replace(/\/$/, "");
const clientId = process.env.OIDC_CLIENT_ID;
const clientSecret = process.env.OIDC_CLIENT_SECRET;
export const authOptions: NextAuthOptions = {
  secret: process.env.NEXTAUTH_SECRET,
  session: { strategy: "jwt", maxAge: 8 * 60 * 60 },
  pages: { signIn: "/signin" },
  providers: issuer && clientId && clientSecret ? [{
    id: "organization",
    name: "Organization account",
    type: "oauth",
    wellKnown: `${issuer}/.well-known/openid-configuration`,
    issuer,
    clientId,
    clientSecret,
    idToken: true,
    checks: ["pkce", "state", "nonce"],
    profile(profile) {
      return {
        id: profile.sub ?? "",
        name: profile.name,
        email: profile.email,
      };
    },
  }] : [],
  callbacks: {
    async signIn({ profile }) {
      return isAllowedOrganizationProfile(profile);
    },
    async jwt({ token, profile }) {
      if (profile) token.qednetAuthorized = isAllowedOrganizationProfile(profile);
      return token;
    },
  },
};
