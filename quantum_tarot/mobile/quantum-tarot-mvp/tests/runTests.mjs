/**
 * Test Runner for LunatiQ (Node.js compatible)
 * Uses dynamic imports to load ES6 modules
 */

async function main() {
  try {
    // Dynamic imports for ES6 modules
    const { runAllTests } = await import('./testLunatiQ.js');

    // Run the test suite
    runAllTests();

  } catch (error) {
    console.error('\n✗ Failed to run tests:');
    console.error(error);
    process.exit(1);
  }
}

main();
