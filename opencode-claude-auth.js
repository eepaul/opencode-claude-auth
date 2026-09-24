// Entry point: re-exports the compiled plugin from dist/. Only the default
// export is exposed: it carries setup() for OpenCode 2.x and server() for
// OpenCode 1.x, and an extra named plugin export would load twice under V1.
export { default } from "./dist/index.js"
