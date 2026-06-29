use std::fs;
use std::thread;

pub struct HardwareConfig {
    pub l2_cache_size: usize,
    pub tlb_entries: usize,
    pub cache_line_size: usize,
    pub num_cpus: usize,
}

pub fn get_hardware_config() -> HardwareConfig {
    // Basic dynamic implementation.
    // Attempt to parse /sys/devices/system/cpu/cpu0/cache/index2/size for L2 cache
    // and fallback to sensible defaults if running on MacOS/Windows or if files are inaccessible.
    let l2_cache_size =
        fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index2/size")
            .ok()
            .and_then(|s| {
                let s = s.trim();
                if s.ends_with('K') {
                    s[..s.len() - 1].parse::<usize>().ok().map(|k| k * 1024)
                } else if s.ends_with('M') {
                    s[..s.len() - 1]
                        .parse::<usize>()
                        .ok()
                        .map(|m| m * 1024 * 1024)
                } else {
                    s.parse::<usize>().ok()
                }
            })
            .unwrap_or(256 * 1024); // Fallback: 256KB

    let cache_line_size = fs::read_to_string(
        "/sys/devices/system/cpu/cpu0/cache/index2/coherency_line_size",
    )
    .ok()
    .and_then(|s| s.trim().parse::<usize>().ok())
    .unwrap_or(64);

    let tlb_entries = 64; // TLB entries are harder to parse directly from sysfs, keeping fallback

    HardwareConfig {
        l2_cache_size,
        tlb_entries,
        cache_line_size,
        num_cpus: thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_hardware_config() {
        let config = get_hardware_config();
        assert!(config.l2_cache_size > 0);
        assert!(config.tlb_entries > 0);
        assert!(config.cache_line_size >= 64);
        assert!(config.num_cpus > 0);
    }
}
