const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Library module
    const ason_mod = b.addModule("ason", .{
        .root_source_file = b.path("src/ason.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Examples
    const example_names = [_][]const u8{ "basic", "complex", "bench", "cross_compat", "bench_test" };
    const example_srcs = [_][]const u8{ "examples/basic.zig", "examples/complex.zig", "examples/bench.zig", "examples/cross_compat.zig", "tests/bench_test.zig" };

    inline for (0..example_names.len) |idx| {
        const mod = b.createModule(.{
            .root_source_file = b.path(example_srcs[idx]),
            .target = target,
            .optimize = optimize,
        });
        mod.addImport("ason", ason_mod);

        const exe = b.addExecutable(.{
            .name = example_names[idx],
            .root_module = mod,
        });
        b.installArtifact(exe);

        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());
        const run_step = b.step(example_names[idx], b.fmt("Run {s} example", .{example_names[idx]}));
        run_step.dependOn(&run_cmd.step);
    }

    // Tests
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/ason.zig"),
        .target = target,
        .optimize = optimize,
    });
    const lib_tests = b.addTest(.{
        .root_module = test_mod,
    });
    const run_tests = b.addRunArtifact(lib_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // Cross-compat tests
    const cross_mod = b.createModule(.{
        .root_source_file = b.path("tests/cross_compat_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    cross_mod.addImport("ason", ason_mod);
    const cross_tests = b.addTest(.{
        .root_module = cross_mod,
    });
    const run_cross = b.addRunArtifact(cross_tests);
    test_step.dependOn(&run_cross.step);
}
