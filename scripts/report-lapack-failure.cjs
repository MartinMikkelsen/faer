// Called only by the scheduled/manual workflow; offline tests mock every API.
module.exports = async ({github, context, result, server}) => {
  if (!['failure', 'cancelled'].includes(result)) return;
  const repo = context.repo;
  const label = 'scheduled-lapack-validation';
  const title = 'Scheduled LAPACK numerical validation is failing';
  try {
    await github.rest.issues.getLabel({...repo, name: label});
  } catch (error) {
    if (error.status !== 404) throw error;
    await github.rest.issues.createLabel({
      ...repo, name: label, color: 'B60205',
      description: 'Failures in the scheduled bounded LAPACK numerical validation',
    });
  }
  const issues = await github.paginate(github.rest.issues.listForRepo, {
    ...repo, state: 'open', labels: label, per_page: 100,
  });
  const issue = issues.find(item => !item.pull_request && item.title === title);
  const run = `${server}/${repo.owner}/${repo.repo}/actions/runs/${context.runId}`;
  const body = [
    `Latest failed run: ${run}`,
    `Commit: ${context.sha}`,
    `Validation result: ${result}`,
    '',
    'See the lapack-validation artifact: environment, build timing, exact test selection, per-family logs and summary.json.',
    'Every selected family runs despite earlier numerical failures; known failures remain failures.',
    'Coverage, omissions and local reproduction: docs/lapack-validation.md.',
  ].join('\n');
  if (issue) {
    await github.rest.issues.update({...repo, issue_number: issue.number, body});
  } else {
    await github.rest.issues.create({...repo, title, body, labels: [label]});
  }
};
