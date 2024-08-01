import { type Config } from "drizzle-kit";

import { env } from "nbl/env";

export default {
    schema: "./src/server/db/schema.ts",
    dialect: "postgresql",
    dbCredentials: {
        url: env.DATABASE_URL,
    },
    tablesFilter: ["nbl_site_*"],
} satisfies Config;
