// Open or update ONE GitHub issue for a failing non-required workflow run.
//
// Shared by .github/workflows/ci-extended.yml and ci-deferred.yml (called from
// actions/github-script after actions/checkout):
//
//   const report = require("./scripts/ci_failure_issue.js");
//   await report({ github, context, core }, { title, label, ... });
//
// Deduplication: while an open issue with exactly `title` and `label` exists,
// every later failing run adds a comment to it instead of opening a new one.
// The body links the run, lists every job result and appends each report file
// named `reportName` found under `reportsDir` (downloaded run artifacts).
"use strict";

const fs = require("fs");
const path = require("path");

const BODY_LIMIT = 60000;

function collectReports(dir, reportName) {
  const reports = [];
  const walk = d => {
    if (!fs.existsSync(d)) return;
    for (const entry of fs.readdirSync(d, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      const p = path.join(d, entry.name);
      if (entry.isDirectory()) walk(p);
      else if (entry.name === reportName) reports.push(fs.readFileSync(p, "utf8"));
    }
  };
  walk(dir);
  return reports;
}

function renderBody({ runUrl, eventName, sha, needs, reports, noReport }) {
  const results = Object.entries(needs)
    .map(([job, r]) => `- ${job}: ${r.result}`)
    .join("\n");
  let body = [
    `Run: ${runUrl}`,
    `Event: ${eventName}; commit: ${sha}`,
    "",
    "Job results:",
    results,
    "",
    reports.length ? reports.join("\n") : noReport,
  ].join("\n");
  if (body.length > BODY_LIMIT) {
    body = body.slice(0, BODY_LIMIT) + `\n\n(truncated; full reports are in the run artifacts: ${runUrl})`;
  }
  return body;
}

async function issuesDisabled(core, title, body) {
  const msg = `${title}: GitHub Issues are disabled for this repository, so no issue was opened. ` +
    "Enable Issues (Settings > General > Features) to get one tracked issue per failing workflow; " +
    "the full report is in this job's summary.";
  core.error(msg);
  if (core.summary) {
    await core.summary.addHeading(title, 2).addRaw(body, true).write();
  }
  core.setFailed(msg);
  return null;
}

async function openOrComment({ github, core }, { owner, repo, title, label, labelDescription, intro, body }) {
  try {
    await github.rest.issues.getLabel({ owner, repo, name: label });
  } catch (error) {
    if (error.status !== 404) throw error;
    await github.rest.issues.createLabel({ owner, repo, name: label, color: "b60205", description: labelDescription });
  }

  const open = await github.paginate(github.rest.issues.listForRepo, {
    owner,
    repo,
    state: "open",
    labels: label,
    per_page: 100,
  });
  const existing = open.find(issue => !issue.pull_request && issue.title === title);
  if (existing) {
    await github.rest.issues.createComment({ owner, repo, issue_number: existing.number, body });
    core.info(`Commented on #${existing.number}`);
    return existing.number;
  }
  const created = await github.rest.issues.create({ owner, repo, title, labels: [label], body: intro + "\n\n" + body });
  core.info(`Opened #${created.data.number}`);
  return created.data.number;
}

module.exports = async function reportFailure({ github, context, core }, opts) {
  const { title, label, labelDescription, intro, reportsDir, reportName, noReport } = opts;
  const { owner, repo } = context.repo;
  const runUrl = `${context.serverUrl}/${owner}/${repo}/actions/runs/${context.runId}`;
  const needs = JSON.parse(process.env.NEEDS_JSON || "{}");
  const body = renderBody({
    runUrl,
    eventName: context.eventName,
    sha: context.sha,
    needs,
    reports: reportsDir && reportName ? collectReports(reportsDir, reportName) : [],
    noReport: noReport || "No report artifact was found; see the run log.",
  });

  // Issues can be disabled in the repository settings (the API then answers
  // 410 Gone). Never let the failure signal disappear: put the report in the
  // job summary, annotate the run, and fail this job so the run stays red.
  if (context.payload && context.payload.repository && context.payload.repository.has_issues === false) {
    return issuesDisabled(core, title, body);
  }
  try {
    return await openOrComment({ github, core }, { owner, repo, title, label, labelDescription, intro, body });
  } catch (error) {
    // Issue creation/comment answers 410 Gone when Issues are disabled.
    if (error.status === 410) return issuesDisabled(core, title, body);
    throw error;
  }
};

module.exports.renderBody = renderBody;
module.exports.collectReports = collectReports;
