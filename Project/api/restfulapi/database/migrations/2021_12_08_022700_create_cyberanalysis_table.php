<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateCyberanalysisTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('cyberanalysis', function (Blueprint $table) {
            $table->id();
            $table->float('duration');
            $table->string('protocol_type');
            $table->string('service');
            $table->string('flag');
            $table->float('src_bytes');
            $table->float('dst_bytes');
            $table->float('land');
            $table->float('wrong_fragment');
            $table->float('urgent');
            $table->float('hot');
            $table->float('num_failed_logins');
            $table->float('logged_in');
            $table->float('num_compromised');
            $table->float('root_shell');
            $table->float('su_attempted');
            $table->float('num_root');
            $table->float('num_file_creations');
            $table->float('num_shells');
            $table->float('num_access_files');
            $table->float('num_outbound_cmds');
            $table->float('is_host_login');
            $table->float('is_guest_login');
            $table->float('count');
            $table->float('srv_count');
            $table->float('serror_rate');
            $table->float('srv_serror_rate');
            $table->float('rerror_rate');
            $table->float('srv_rerror_rate');
            $table->float('same_srv_rate');
            $table->float('diff_srv_rate');
            $table->float('srv_diff_host_rate');
            $table->float('dst_host_count');
            $table->float('dst_host_srv_count');
            $table->float('dst_host_same_srv_rate');
            $table->float('dst_host_diff_srv_rate');
            $table->float('dst_host_same_src_port_rate');
            $table->float('dst_host_srv_diff_host_rate');
            $table->float('dst_host_serror_rate');
            $table->float('dst_host_srv_serror_rate');
            $table->float('dst_host_rerror_rate');
            $table->float('dst_host_srv_rerror_rate');
            $table->float('class');
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('cyberanalysis');
    }
}
