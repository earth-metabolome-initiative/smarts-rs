#!/usr/bin/env node

import { cp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");
const activeCorpusDir = path.join(repoRoot, "fuzz", "corpus", "parse_display_loop");
const backupCorpusDir = path.join(
  repoRoot,
  "fuzz",
  "corpus",
  "parse_display_loop.backup-20260330",
);
const parserCorpusDir = path.join(repoRoot, "corpus", "parser");

const fixtureFiles = [
  ["parse-valid-v0.json", "fixture-valid"],
  ["parse-invalid-v0.json", "fixture-invalid"],
];

async function readFixtureSeeds() {
  const seeds = [];

  for (const [fileName, prefix] of fixtureFiles) {
    const filePath = path.join(parserCorpusDir, fileName);
    const entries = JSON.parse(await readFile(filePath, "utf8"));

    for (const entry of entries) {
      seeds.push({
        fileName: `${prefix}-${entry.id}`,
        contents: entry.smarts,
      });
    }
  }

  seeds.push({
    fileName: "regression-overflow-charge-run",
    contents: `[S${"-".repeat(200)}]`,
  });

  return seeds;
}

async function main() {
  await rm(activeCorpusDir, { force: true, recursive: true });
  await mkdir(path.dirname(activeCorpusDir), { recursive: true });
  await cp(backupCorpusDir, activeCorpusDir, { recursive: true });

  const seeds = await readFixtureSeeds();
  for (const seed of seeds) {
    await writeFile(path.join(activeCorpusDir, seed.fileName), seed.contents, "utf8");
  }

  const files = await readdir(activeCorpusDir);
  console.log(
    [
      `rebuilt ${activeCorpusDir}`,
      `copied backup corpus from ${backupCorpusDir}`,
      `added ${seeds.length} deterministic seeds`,
      `total files: ${files.length}`,
    ].join("\n"),
  );
}

await main();
