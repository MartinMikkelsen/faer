const assert = require('node:assert/strict');
const test = require('node:test');
const report = require('./report-lapack-failure.cjs');
const title = 'Scheduled LAPACK numerical validation is failing';
function fixture({missing = false, issues = [], error} = {}) {
  const calls = [];
  const api = {};
  for (const method of ['getLabel', 'createLabel', 'create', 'update']) {
    api[method] = async args => {
      calls.push([method, args]);
      if (method === 'getLabel' && (missing || error)) throw {status: error || 404};
    };
  }
  api.listForRepo = async () => {};
  return {
    calls,
    args: {
      github: {rest: {issues: api}, paginate: async () => issues},
      context: {repo: {owner: 'example', repo: 'fork'}, runId: 7, sha: 'abc'},
      result: 'failure', server: 'https://github.com',
    },
  };
}
test('create one labeled issue on the first failure', async () => {
  const {calls, args} = fixture({missing: true});
  await report(args);
  assert.deepEqual(calls.map(c => c[0]), ['getLabel', 'createLabel', 'create']);
  assert.equal(calls[2][1].title, title);
  assert.match(calls[2][1].body, /actions\/runs\/7/);
});
test('update the matching issue; exclude pull requests and unrelated titles', async () => {
  const {calls, args} = fixture({issues: [
    {number: 1, title, pull_request: {}}, {number: 2, title: 'other'}, {number: 3, title},
  ]});
  await report(args);
  assert.equal(calls.at(-1)[0], 'update');
  assert.equal(calls.at(-1)[1].issue_number, 3);
});
test('cancellation reports and success stays quiet', async () => {
  const {calls, args} = fixture();
  await report({...args, result: 'success'});
  assert.equal(calls.length, 0);
  await report({...args, result: 'cancelled'});
  assert.equal(calls.at(-1)[0], 'create');
});
test('propagate permission/network failures', async () => {
  const {calls, args} = fixture({error: 403});
  await assert.rejects(report(args), error => error.status === 403);
  assert.equal(calls.length, 1);
});
